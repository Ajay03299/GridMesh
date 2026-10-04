"""Transparent known-subtotal cost model. Missing items remain TBD, never zero."""
import math


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{label} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def _band(value, band):
    return value[band] if isinstance(value, dict) else value


def affordability_plan(cfg, band='base', asset_scenario='existing_assets', paying_homes=100,
                       *, conditional_funding=False):
    """Itemised partial cost recovery, with explicit unquoted blockers.

    Upfront recovery is a provision, not a second cash payment. Initial cash and
    recurring cash are separate. Financing replaces equipment amortisation.
    """
    if band not in ('low', 'base', 'high') or asset_scenario not in ('existing_assets', 'new_equipment'):
        raise ValueError('Unknown planning band or asset scenario')
    if not isinstance(paying_homes, int) or isinstance(paying_homes, bool) or not 0 < paying_homes <= cfg['community_homes']:
        raise ValueError('Participation must be 1..community_homes')
    recovery = cfg['capital_recovery']
    if recovery not in ('replacement', 'financing'):
        raise ValueError('Select replacement OR financing')
    months = _number(cfg['setup_recovery_months'], 'setup recovery months', positive=True)
    life = _number(cfg['backup_life_months'], 'asset life months', positive=True)
    fx = _number(_band(cfg['usd_to_inr'], band), 'planning exchange conversion', positive=True)
    contingency = _number(_band(cfg['contingency_fraction'], band), 'contingency')
    if contingency > 1 or cfg['commercial_margin_fraction'] != 0:
        raise ValueError('Contingency must be <=100%; model supports cost recovery, not selling prices')
    names = set()
    details, unknown = [], {'incremental': [], 'baseline_operating': [], 'new_asset': []}
    subtotal = {scope: {'upfront': 0., 'monthly': 0., 'recovery': 0.} for scope in unknown}
    for item_cfg in cfg['items']:
        name, scope, period = item_cfg['name'], item_cfg['scope'], item_cfg['period']
        if name in names:
            raise ValueError('Duplicate cost item')
        names.add(name)
        if scope not in subtotal or period not in ('upfront', 'monthly', 'usage'):
            raise ValueError('Invalid cost scope or period')
        if item_cfg['source'] not in cfg['sources'] or not item_cfg.get('payer'):
            raise ValueError('Every cost needs source/assumption and payer')
        if scope == 'new_asset' and asset_scenario == 'existing_assets':
            continue
        if name == 'existing_asset_access' and asset_scenario == 'new_equipment':
            continue
        if item_cfg.get('recovery') and item_cfg['recovery'] != recovery:
            continue
        quantity, rate, tax = [_band(item_cfg[key], band) for key in ('quantity', 'rate', 'tax')]
        for value, label in ((quantity, 'quantity'), (rate, 'rate'), (tax, 'tax')):
            if value is not None:
                _number(value, f'{name} {label}')
        if tax is not None and tax > 1:
            raise ValueError('Tax fraction must be <=100%')
        conversion = _number(item_cfg.get('unit_conversion', 1), 'unit conversion', positive=True)
        currency = item_cfg['currency']
        if currency not in ('USD', 'INR'):
            raise ValueError('Currency must be USD or INR')
        missing = [key for key, value in (('quantity', quantity), ('rate', rate), ('tax', tax)) if value is None]
        net = None if quantity is None or rate is None else quantity * rate * conversion * (fx if currency == 'USD' else 1)
        gross = None if missing else net * (1 + tax)
        recovery_monthly = None if gross is None else (gross / (life if scope == 'new_asset' else months) if period == 'upfront' else 0.)
        if scope == 'new_asset' and recovery == 'financing':
            recovery_monthly = 0. # financed principal already belongs in monthly repayment
        if missing:
            unknown[scope].append(name + ': ' + '/'.join(missing))
        if gross is not None:
            bucket = 'upfront' if period == 'upfront' else 'monthly'
            subtotal[scope][bucket] += gross
            subtotal[scope]['recovery'] += recovery_monthly
        details.append(dict(name=name, scope=scope, period=period, quantity=quantity,
                            rate=rate, currency=currency, net_inr=net, tax_fraction=tax,
                            gross_inr=gross, recovery_monthly_inr=recovery_monthly,
                            payer=item_cfg['payer'], source=item_cfg['source'], missing=missing))
    inc = subtotal['incremental']
    reserve = contingency * (inc['monthly'] + inc['recovery'])
    covered = inc['monthly'] + inc['recovery'] + reserve
    capital = subtotal['new_asset']['monthly'] + subtotal['new_asset']['recovery']
    operating_partial = covered + subtotal['baseline_operating']['monthly'] + capital
    funding = cfg['funding']
    sponsor = _number(funding['sponsor_monthly_inr'], 'sponsor payment')
    if sponsor and not funding.get('sponsor_name'):
        raise ValueError('Funding needs a named payer')
    confirmed = funding.get('sponsor_confirmed') is True
    applied = sponsor if confirmed or conditional_funding else 0.
    remaining = max(0., covered - applied)
    payment = funding.get('proposed_household_payment_inr')
    cash_shortfall = None if payment is None else max(0., covered - applied - paying_homes * _number(payment, 'household payment'))
    delta = cfg.get('operating_change_monthly_inr')
    if delta is not None and (isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta)):
        raise ValueError('Operating cost change must be finite or TBD')
    inc_complete = not unknown['incremental'] and not unknown['new_asset'] and delta is not None
    # Post-deployment consumed-energy/O&M costs already include any change.
    # Do not add the incremental delta again to total operating cost.
    total_complete = not unknown['incremental'] and not unknown['baseline_operating'] and not unknown['new_asset']
    full_incremental = None if not inc_complete else covered + capital + delta
    if full_incremental is not None and full_incremental < 0:
        raise ValueError('Net savings exceed costs; requires a separate benefit analysis')
    return dict(band=band, asset_scenario=asset_scenario, paying_homes=paying_homes,
                details=details, known_upfront_incremental_inr=inc['upfront'],
                known_monthly_incremental_cash_inr=inc['monthly'], setup_recovery_monthly_inr=inc['recovery'],
                contingency_monthly_inr=reserve, covered_service_monthly_inr=covered,
                covered_cost_recovery_per_home_inr=remaining / paying_homes,
                covered_total_operating_monthly_inr=operating_partial,
                total_operating_monthly_inr=operating_partial if total_complete else None,
                full_incremental_monthly_inr=full_incremental,
                full_incremental_contribution_inr=max(0., full_incremental-applied)/paying_homes if inc_complete else None,
                operating_change_monthly_inr=delta,
                incremental_complete=inc_complete, total_complete=total_complete, unknown_items=unknown,
                sponsor_applied_inr=applied, sponsor_surplus_inr=max(0., applied-covered),
                conditional_funding=bool(applied and not confirmed), covered_shortfall_inr=cash_shortfall,
                shortfall_payer='association reserves or named sponsor, unconfirmed',
                commercial_margin_fraction=0, price_validated=False,
                known_initial_capital_cash_inr=subtotal['new_asset']['upfront'],
                initial_capital_cash_inr=None if any(x.startswith('new_backup_equipment:') for x in unknown['new_asset']) else subtotal['new_asset']['upfront'],
                initial_capital_complete=not any(x.startswith('new_backup_equipment:') for x in unknown['new_asset']))


def full_service(cfg):
    """Known subtotals are never labelled complete when an applicable item is TBD.

    Financing pays capital recovery instead of a replacement provision, avoiding
    automatic double-counting. Tax rates belong to each quote, not a guessed GST.
    """
    rows = []
    mode = cfg['capital_recovery']
    if mode not in ('replacement', 'financing'):
        raise ValueError('capital_recovery must be replacement or financing')
    for scenario in ('existing_assets', 'new_equipment'):
        costs = {'upfront': [], 'monthly': []}
        unknown = []
        for entry in cfg['items']:
            if scenario not in entry['applies_to']:
                continue
            if entry.get('recovery') and entry['recovery'] != mode:
                continue
            period = entry['period']
            if period not in costs or not entry.get('payer') or not entry.get('source'):
                raise ValueError('Cost item requires valid period, payer and source')
            amount, tax = entry['inr'], entry['tax_fraction']
            for value in (amount, tax):
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value < 0):
                    raise ValueError('Cost/tax must be finite nonnegative or TBD')
            if amount is None or tax is None:
                unknown.append(entry['name'])
            if amount is not None:
                # Known base amounts remain in the PARTIAL subtotal; missing tax
                # still blocks any complete price or break-even claim.
                costs[period].append(amount*(1+(tax if tax is not None else 0)))
        for homes in cfg['participation']:
            if not isinstance(homes, int) or isinstance(homes, bool) or homes <= 0:
                raise ValueError('Paying homes must be positive integer')
            for extra in cfg['monthly_driver_additions_inr']:
                if not math.isfinite(extra) or extra < 0:
                    raise ValueError('Driver addition must be finite nonnegative')
                upfront, monthly = sum(costs['upfront']), sum(costs['monthly'])+extra
                rows.append(dict(scenario=scenario,paying_homes=homes,
                    hypothetical_driver_addition_inr=extra,
                    known_upfront_subtotal_inr=upfront,known_monthly_subtotal_inr=monthly,
                    known_monthly_per_home_inr=monthly/homes,
                    full_service_price_inr=None if unknown else monthly/homes,
                    break_even_monthly_fee_inr=None if unknown else monthly/homes,
                    complete=not unknown,unknown_items='; '.join(unknown),
                    status='Assumed complete estimate, not validated' if not unknown else 'TBD: partial subtotal, not total affordability'))
    return rows


def item(cfg, name):
    value = cfg[name]['value']
    if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
        raise ValueError(f'{name} must be finite nonnegative or null/TBD')
    if not cfg[name].get('source'):
        raise ValueError(f'{name} requires an assumption/source label')
    return value


def scenarios(cfg):
    common = ['software_hosting_monthly', 'operator_labour_monthly', 'maintenance_monthly',
              'connectivity_monthly', 'sensors_gateways_monthly']
    values = {k:item(cfg, k) for k in common}
    mwh, price = item(cfg, 'backup_energy_monthly_mwh'), item(cfg, 'backup_energy_inr_per_kwh')
    values['backup_energy_monthly'] = None if mwh is None or price is None else 1000*mwh*price
    access = item(cfg, 'existing_asset_access_monthly')
    capex, life = item(cfg, 'new_asset_capex'), item(cfg, 'new_asset_life_months')
    if life == 0:
        raise ValueError('Asset life must be positive')
    finance = item(cfg, 'new_asset_financing_monthly')
    if finance is not None and capex is not None and life is not None:
        raise ValueError('Select financing OR straight-line equipment recovery, not both')
    rows = []
    for name, extra in [('existing_asset_incremental', {'asset_access':access}),
                        ('new_asset_deployment', {'replacement_provision':None if finance is not None or capex is None or life is None else capex/life,
                                                   'financing':finance})]:
        costs = values | extra
        known = sum(v for v in costs.values() if v is not None)
        unknown = [k for k,v in costs.items() if v is None]
        for homes in cfg['participation']:
            if not isinstance(homes, int) or homes <= 0:
                raise ValueError('Paying homes must be a positive integer')
            for addition in cfg.get('monthly_cost_additions', [0]):
                if not math.isfinite(addition) or addition < 0:
                    raise ValueError('Sensitivity additions must be nonnegative')
                rows.append(dict(scenario=name, paying_homes=homes,
                    assumed_additional_monthly_inr=addition, known_monthly_subtotal_inr=known+addition,
                    known_subtotal_per_paying_home_inr=(known+addition)/homes,
                    complete=not unknown, unknown_items='; '.join(unknown),
                    status='Complete assumed estimate, not validated' if not unknown else 'INCOMPLETE known subtotal, not an all-in price'))
    return rows
